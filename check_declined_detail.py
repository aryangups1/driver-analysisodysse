"""
Check what data is available on declined/non-responded trips,
and do a deeper Zone 3 breakdown.
"""
import psycopg2
import pandas as pd
from zones import parse_dms, assign_zone
from config import TOP_DRIVER_IDS, DRIVER_NAMES

conn = psycopg2.connect(
    host='dev-odysse.postgres.database.azure.com', port=5432,
    dbname='unifieddwh', user='odysys', password='Odysys@2026', sslmode='require'
)
IDS = TOP_DRIVER_IDS

# ── 1. What columns are populated on declined trips? ─────────────────────────
print("=== DECLINED TRIP DATA COMPLETENESS ===")
cur = conn.cursor()
cur.execute("""
    SELECT
        COUNT(*) as total,
        COUNT(trip_price_in_pound) as has_fare,
        COUNT(distance_in_miles) as has_distance,
        COUNT(dropoff_address) as has_dropoff_addr,
        COUNT(dropoff_latlong) as has_dropoff_coord,
        COUNT(pickup_address) as has_pickup_addr,
        COUNT(pickup_lat_long) as has_pickup_coord,
        COUNT(pickedup_trip_datetime) as has_pickup_dt,
        COUNT(trip_booking_datetime) as has_booking_dt
    FROM rep_fact_trips
    WHERE dim_driver_id = ANY(%s)
      AND status IN ('Driver did not respond', 'Driver rejected')
""", (IDS,))
r = cur.fetchone()
cols = [d[0] for d in cur.description]
print(f"  Total declined: {r[0]:,}")
for c, v in zip(cols[1:], r[1:]):
    pct = v/r[0]*100 if r[0] else 0
    print(f"  {c:<28} {v:>7,}  ({pct:.0f}%)")

# ── 2. Sample declined trips — what do they look like? ───────────────────────
print("\n=== SAMPLE DECLINED TRIPS (with destination) ===")
cur.execute("""
    SELECT dim_driver_id, driver_full_name, status,
           trip_booking_datetime,
           pickup_address, dropoff_address,
           pickup_lat_long, dropoff_latlong,
           trip_price_in_pound, distance_in_miles, trips_hr
    FROM rep_fact_trips
    WHERE dim_driver_id = ANY(%s)
      AND status IN ('Driver did not respond', 'Driver rejected')
      AND dropoff_address IS NOT NULL AND dropoff_address != ''
      AND trip_price_in_pound IS NOT NULL
    ORDER BY dim_driver_id, trip_booking_datetime DESC
    LIMIT 20
""", (IDS,))
rows = cur.fetchall()
print(f"  Declined trips WITH destination info: checking...")
for r in rows[:10]:
    name = DRIVER_NAMES.get(r[0], str(r[0]))
    pu = (r[4] or '')[:35]
    do = (r[5] or '')[:35]
    print(f"  {name:<22}  £{r[8] or 0:.2f}  {r[9] or 0:.1f}mi  hr={r[10]}")
    print(f"    FROM: {pu}")
    print(f"      TO: {do}")

# ── 3. Accepted vs declined: fare and distance comparison (Bolt only) ─────────
print("\n=== ACCEPTED vs DECLINED — FARE & DISTANCE ===")
cur.execute("""
    SELECT
        CASE WHEN status IN ('completed','Finished') THEN 'Accepted'
             WHEN status IN ('Driver did not respond','Driver rejected') THEN 'Declined'
        END as outcome,
        ROUND(AVG(trip_price_in_pound)::numeric,2) as avg_fare,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY trip_price_in_pound)::numeric,2) as median_fare,
        ROUND(AVG(distance_in_miles)::numeric,2) as avg_dist,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY distance_in_miles)::numeric,2) as median_dist,
        COUNT(*) as trips
    FROM rep_fact_trips
    WHERE dim_driver_id = ANY(%s)
      AND status IN ('completed','Finished','Driver did not respond','Driver rejected')
      AND trip_price_in_pound IS NOT NULL AND trip_price_in_pound > 0
    GROUP BY outcome
""", (IDS,))
print(f"  {'Outcome':<12} {'Avg Fare':>10} {'Med Fare':>10} {'Avg Dist':>10} {'Med Dist':>10} {'Count':>8}")
for r in cur.fetchall():
    print(f"  {r[0]:<12} £{r[1]:>9} £{r[2]:>9} {r[3]:>9}mi {r[4]:>9}mi {r[5]:>8,}")

# ── 4. Accepted vs declined: dropoff zone comparison ─────────────────────────
print("\n=== ACCEPTED vs DECLINED — DROPOFF ZONE (where were they going?) ===")
cur.execute("""
    SELECT status, dropoff_latlong, trip_price_in_pound, distance_in_miles, trips_hr
    FROM rep_fact_trips
    WHERE dim_driver_id = ANY(%s)
      AND status IN ('completed','Finished','Driver did not respond','Driver rejected')
      AND dropoff_latlong IS NOT NULL AND dropoff_latlong != ''
      AND trip_price_in_pound > 0
    LIMIT 50000
""", (IDS,))
rows = cur.fetchall()
records = []
for r in rows:
    dlat, dlon = parse_dms(r[1])
    dz = assign_zone(dlat, dlon)
    outcome = 'Accepted' if r[0] in ('completed','Finished') else 'Declined'
    records.append({'outcome': outcome, 'dropoff_zone': dz,
                    'fare': r[2], 'distance': r[3], 'hour': r[4]})

df = pd.DataFrame(records).dropna(subset=['dropoff_zone'])
df['dropoff_zone'] = df['dropoff_zone'].astype(int)

pivot = df.groupby(['outcome','dropoff_zone']).size().unstack(fill_value=0)
pivot_pct = (pivot.div(pivot.sum(axis=1), axis=0) * 100).round(1)
print("  Dropoff zone distribution (%):")
print(pivot_pct.to_string())

print("\n  Avg fare by dropoff zone and outcome:")
fare_dz = df.groupby(['outcome','dropoff_zone'])['fare'].mean().unstack().round(2)
print(fare_dz.to_string())

# ── 5. Zone 3 deep dive ───────────────────────────────────────────────────────
print("\n\n=== ZONE 3 DEEP DIVE ===")
cur.execute("""
    SELECT pickup_lat_long, dropoff_latlong, pickup_address,
           trip_price_in_pound, distance_in_miles, pob_duration_in_min,
           pickup_duration_in_min, trips_hr, status,
           pickedup_trip_datetime, dropoff_trip_datetime
    FROM rep_fact_trips
    WHERE dim_driver_id = ANY(%s)
      AND status IN ('completed','Finished')
      AND distance_in_miles <= 60
      AND pickup_lat_long IS NOT NULL AND pickup_lat_long != ''
    ORDER BY pickedup_trip_datetime
""", (IDS,))
rows = cur.fetchall()
z3_records = []
for r in rows:
    plat, plon = parse_dms(r[0])
    pz = assign_zone(plat, plon)
    if pz == 3:
        dlat, dlon = parse_dms(r[1])
        dz = assign_zone(dlat, dlon)
        z3_records.append({
            'pickup_address': r[2], 'fare': r[3], 'distance': r[4],
            'ride_mins': r[5], 'pickup_mins': r[6], 'hour': r[7],
            'dropoff_zone': dz, 'pickup_lat': plat, 'pickup_lon': plon,
        })

z3 = pd.DataFrame(z3_records)
print(f"  Zone 3 accepted trips: {len(z3)}")
print(f"\n  By hour of day — which hours are worth it?")
z3_hour = z3.groupby('hour').agg(
    trips=('fare','count'),
    avg_fare=('fare','mean'),
    avg_dist=('distance','mean'),
    avg_ride=('ride_mins','mean'),
    avg_pickup_wait=('pickup_mins','mean'),
).round(1)
# Estimate true RPH for each hour bucket using avg fare and total time
for h, row in z3_hour.iterrows():
    # rough true RPH using avg ride mins only (no inter-trip wait here)
    rph_approx = (row.avg_fare / (row.avg_ride / 60)) if row.avg_ride > 0 else 0
    print(f"    {h:02d}:00  trips={int(row.trips):<4} fare=£{row.avg_fare:.2f}  dist={row.avg_dist:.1f}mi  ride={row.avg_ride:.0f}min  raw_rph=£{rph_approx:.0f}")

print(f"\n  Most common Zone 3 pickup areas:")
if 'pickup_address' in z3.columns:
    # Extract borough/area from address
    z3['area'] = z3['pickup_address'].str.extract(r',\s*([^,]+),\s*(?:London|Greater London)', expand=False)
    area_counts = z3['area'].value_counts().head(15)
    for area, count in area_counts.items():
        if area and str(area) != 'nan':
            avg_f = z3[z3['area']==area]['fare'].mean()
            print(f"    {str(area):<35} {count:>4} trips  avg £{avg_f:.2f}")

print(f"\n  Zone 3 wait time distribution:")
print(f"    (Need inter-trip gap — see main dashboard Zone Analysis page)")
print(f"    Ride time p25={z3['ride_mins'].quantile(.25):.0f}min  median={z3['ride_mins'].median():.0f}min  p75={z3['ride_mins'].quantile(.75):.0f}min  p90={z3['ride_mins'].quantile(.9):.0f}min")

print(f"\n  Where do Zone 3 trips drop off?")
dz_counts = z3['dropoff_zone'].value_counts().sort_index()
for z_val, cnt in dz_counts.items():
    if pd.notna(z_val):
        print(f"    Dropoff Zone {int(z_val)}: {cnt} trips ({cnt/len(z3)*100:.0f}%)")

cur.close()
conn.close()
print("\nDone.")
