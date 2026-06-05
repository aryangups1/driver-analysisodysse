"""
Pull everything and surface the key patterns across the top 10 drivers.
Run this once to get the raw findings — the summary page will show these live.
"""
import psycopg2
import pandas as pd
import numpy as np
from zones import enrich_zones, calc_true_rph
from config import TOP_DRIVER_IDS, DRIVER_NAMES

conn = psycopg2.connect(
    host='dev-odysse.postgres.database.azure.com', port=5432,
    dbname='unifieddwh', user='odysys', password='Odysys@2026', sslmode='require'
)
IDS = TOP_DRIVER_IDS

# ── 1. Performance summary ────────────────────────────────────────────────────
perf = pd.read_sql("""
    SELECT dim_driver_id,
           MAX(driver_full_name) as name,
           COUNT(DISTINCT driver_performance_date) as days,
           SUM(number_of_finished_rides) as rides,
           ROUND(SUM(online_time_in_hrs)::numeric,1) as online_hrs,
           ROUND((SUM(revenue)/NULLIF(SUM(online_time_in_hrs),0))::numeric,2) as rph,
           ROUND(AVG(utilisation_percent)::numeric,1) as util,
           ROUND(AVG(NULLIF(total_acceptance_rate_percent,0))::numeric,1) as accept
    FROM rep_fact_driver_performance
    WHERE dim_driver_id = ANY(%s) AND online_time_in_hrs > 0
    GROUP BY dim_driver_id ORDER BY rph DESC
""", conn, params=(IDS,))

# ── 2. Fleet baseline ─────────────────────────────────────────────────────────
fleet = pd.read_sql("""
    SELECT ROUND(AVG(revenue/NULLIF(online_time_in_hrs,0))::numeric,2) as rph,
           ROUND(AVG(utilisation_percent)::numeric,1) as util,
           ROUND(AVG(NULLIF(total_acceptance_rate_percent,0))::numeric,1) as accept
    FROM rep_fact_driver_performance
    WHERE dim_driver_id NOT IN %s AND online_time_in_hrs > 0 AND revenue > 0
""", conn, params=(tuple(IDS),))

# ── 3. Trip-level data with zones ─────────────────────────────────────────────
trips_raw = pd.read_sql("""
    SELECT dim_driver_id, driver_full_name,
           pickup_lat_long, dropoff_latlong,
           trip_price_in_pound, distance_in_miles,
           pob_duration_in_min, pickup_duration_in_min,
           trips_hr, source,
           pickedup_trip_datetime, dropoff_trip_datetime
    FROM rep_fact_trips
    WHERE dim_driver_id = ANY(%s)
      AND status IN ('completed','Finished')
      AND pickup_lat_long IS NOT NULL AND pickup_lat_long != ''
      AND dropoff_latlong IS NOT NULL AND dropoff_latlong != ''
      AND distance_in_miles <= 60
    ORDER BY dim_driver_id, pickedup_trip_datetime
""", conn, params=(IDS,))

print("Enriching zones...")
zoned = enrich_zones(trips_raw)
zoned = calc_true_rph(zoned)
zoned["display_name"] = zoned["dim_driver_id"].map(DRIVER_NAMES)
zoned["pickup_zone"] = zoned["pickup_zone"].astype(int)
zoned["dropoff_zone"] = zoned["dropoff_zone"].astype(int)

print(f"Total valid trips: {len(zoned)}")

# ── 4. Hourly patterns ────────────────────────────────────────────────────────
hourly = zoned.groupby("trips_hr").agg(
    trips=("trip_price_in_pound","count"),
    avg_fare=("trip_price_in_pound","mean"),
    avg_true_rph=("true_rph","mean"),
).reset_index()
top_hours = hourly.nlargest(6, "avg_true_rph")["trips_hr"].sort_values().tolist()
peak_volume_hours = hourly.nlargest(6, "trips").index.tolist()

# ── 5. Zone efficiency ────────────────────────────────────────────────────────
zone_eff = zoned.groupby("pickup_zone").agg(
    trips=("trip_price_in_pound","count"),
    avg_fare=("trip_price_in_pound","mean"),
    avg_wait=("gap_mins","mean"),
    avg_ride=("pob_duration_in_min","mean"),
    true_rph=("true_rph","mean"),
    avg_dropoff=("dropoff_zone","mean"),
    fare_per_mile=("trip_price_in_pound", lambda x: (x / zoned.loc[x.index,"distance_in_miles"].replace(0,np.nan)).mean()),
).round(2)
best_zone = zone_eff["true_rph"].idxmax()
worst_zone = zone_eff["true_rph"].idxmin()

# ── 6. Per-driver peak hours ──────────────────────────────────────────────────
driver_hours = zoned.groupby(["display_name","trips_hr"])["trip_price_in_pound"].agg(["count","mean"]).reset_index()
driver_hours.columns = ["driver","hour","trips","avg_fare"]
driver_peak = driver_hours.sort_values(["driver","trips"], ascending=[True,False]).groupby("driver").first()

# ── 7. Shift timing ───────────────────────────────────────────────────────────
shift = pd.read_sql("""
    SELECT dim_driver_id, DATE(pickedup_trip_datetime) as d,
           MIN(trips_hr) as start_hr, MAX(trips_hr) as end_hr
    FROM rep_fact_trips
    WHERE dim_driver_id = ANY(%s)
      AND status IN ('completed','Finished')
      AND trips_hr IS NOT NULL AND distance_in_miles <= 60
    GROUP BY dim_driver_id, DATE(pickedup_trip_datetime)
    HAVING COUNT(*) >= 4
""", conn, params=(IDS,))
shift["display_name"] = shift["dim_driver_id"].map(DRIVER_NAMES)
shift_summary = shift.groupby("display_name").agg(
    avg_start=("start_hr","mean"), avg_end=("end_hr","mean")
).round(1)

# ── 8. Platform split ─────────────────────────────────────────────────────────
platform = zoned.groupby(["display_name","source"]).size().reset_index(name="trips")
platform_pct = platform.copy()
totals = platform_pct.groupby("display_name")["trips"].transform("sum")
platform_pct["pct"] = (platform_pct["trips"] / totals * 100).round(0)

# ── 9. Session strategy ───────────────────────────────────────────────────────
zoned["trip_date"] = pd.to_datetime(zoned["pickedup_trip_datetime"]).dt.date
def classify_session(g):
    g = g.sort_values("pickedup_trip_datetime")
    if len(g) < 4: return None
    h = len(g) // 2
    return "big_first" if g.iloc[:h]["distance_in_miles"].mean() > g.iloc[h:]["distance_in_miles"].mean() * 1.3 else \
           "small_first" if g.iloc[h:]["distance_in_miles"].mean() > g.iloc[:h]["distance_in_miles"].mean() * 1.3 else "mixed"
session_labels = zoned.groupby(["display_name","trip_date"]).apply(classify_session).dropna()
session_dist = session_labels.groupby(["display_name", session_labels]).size().unstack(fill_value=0)
if not session_dist.empty:
    session_dist_pct = (session_dist.div(session_dist.sum(axis=1), axis=0) * 100).round(0)

# ── 10. Wait time by zone ─────────────────────────────────────────────────────
wait_zone = zoned.groupby("pickup_zone")["gap_mins"].agg(["mean","median"]).round(1)

# ────────────────────── PRINT FINDINGS ───────────────────────────────────────
f = fleet.iloc[0]
print("\n" + "="*65)
print("PATTERN FINDINGS")
print("="*65)

print(f"\n[A] TOP 10 vs FLEET")
print(f"  Fleet avg RPH:        £{f.rph}/hr")
print(f"  Top 10 avg RPH:       £{perf['rph'].mean():.2f}/hr  (+{perf['rph'].mean()-float(f.rph):.2f})")
print(f"  Fleet utilisation:    {f.util}%")
print(f"  Top 10 utilisation:   {perf['util'].mean():.1f}%  (+{perf['util'].mean()-float(f.util):.1f}%)")
print(f"  Fleet accept rate:    {f.accept}%")
print(f"  Top 10 accept rate:   {perf['accept'].mean():.1f}%  ({perf['accept'].mean()-float(f.accept):+.1f}%)")

print(f"\n[B] ZONE EFFICIENCY (True RPH incl. wait time)")
for z, row in zone_eff.iterrows():
    print(f"  Zone {z}: True RPH £{row.true_rph:.0f}  |  avg fare £{row.avg_fare:.2f}  |  avg wait {row.avg_wait:.0f}min  |  avg ride {row.avg_ride:.0f}min  |  avg dropoff Z{row.avg_dropoff:.1f}")

print(f"\n  -> Best zone by True RPH: Zone {best_zone}")
print(f"  -> Worst zone by True RPH: Zone {worst_zone}")

print(f"\n[C] PEAK HOURS BY TRUE RPH")
print(f"  Top 6 hours:  {[f'{h:02d}:00' for h in top_hours]}")
print(f"\n  Per driver peak hour:")
for drv, row in driver_peak.iterrows():
    print(f"    {drv:<28}  peak: {int(row.hour):02d}:00  ({int(row.trips)} trips, avg £{row.avg_fare:.2f})")

print(f"\n[D] TYPICAL SHIFT TIMES")
for drv, row in shift_summary.iterrows():
    print(f"  {drv:<28}  starts ~{row.avg_start:.0f}:00  ends ~{row.avg_end:.0f}:00")

print(f"\n[E] WAIT TIMES BY PICKUP ZONE")
for z, row in wait_zone.iterrows():
    print(f"  Zone {z}: mean {row['mean']} min  median {row['median']} min")

print(f"\n[F] PLATFORM SPLIT")
for drv in platform_pct["display_name"].unique():
    d = platform_pct[platform_pct.display_name==drv][["source","pct"]].set_index("source")["pct"].to_dict()
    print(f"  {drv:<28}  {d}")

conn.close()
print("\nDone.")
