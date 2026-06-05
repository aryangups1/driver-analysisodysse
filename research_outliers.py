"""
Investigate outlier trips for the top 10 drivers.
Looks at coordinate extremes, distance outliers, and fare outliers.
"""
import psycopg2
import pandas as pd
from zones import parse_dms, assign_zone
from config import TOP_DRIVER_IDS, DRIVER_NAMES

conn = psycopg2.connect(
    host='dev-odysse.postgres.database.azure.com',
    port=5432, dbname='unifieddwh',
    user='odysys', password='Odysys@2026', sslmode='require'
)

IDS = TOP_DRIVER_IDS

trips = pd.read_sql("""
    SELECT dim_driver_id, driver_full_name,
           pickup_lat_long, dropoff_latlong,
           pickup_address, dropoff_address,
           trip_price_in_pound, distance_in_miles,
           pob_duration_in_min, pickedup_trip_datetime, source
    FROM rep_fact_trips
    WHERE dim_driver_id = ANY(%s)
      AND status = 'completed'
      AND pickup_lat_long IS NOT NULL AND pickup_lat_long != ''
      AND dropoff_latlong IS NOT NULL AND dropoff_latlong != ''
""", conn, params=(IDS,))

# Parse coordinates
coords_p = trips["pickup_lat_long"].apply(parse_dms)
coords_d = trips["dropoff_latlong"].apply(parse_dms)
trips["plat"] = [c[0] for c in coords_p]
trips["plon"] = [c[1] for c in coords_p]
trips["dlat"] = [c[0] for c in coords_d]
trips["dlon"] = [c[1] for c in coords_d]
trips["pickup_zone"]  = trips.apply(lambda r: assign_zone(r.plat, r.plon), axis=1)
trips["dropoff_zone"] = trips.apply(lambda r: assign_zone(r.dlat, r.dlon), axis=1)
trips["display_name"] = trips["dim_driver_id"].map(DRIVER_NAMES)

print(f"Total trips loaded: {len(trips)}")
print(f"Parsed coords OK:   {trips['plat'].notna().sum()}")

# ── 1. Coordinate range extremes ─────────────────────────────────────────────
print("\n[1] COORDINATE EXTREMES (pickup lat/lon range)")
print(f"  Lat range: {trips['plat'].min():.4f} to {trips['plat'].max():.4f}  (London ~51.3 to 51.7)")
print(f"  Lon range: {trips['plon'].min():.4f} to {trips['plon'].max():.4f}  (London ~-0.52 to 0.33)")

# ── 2. Trips outside Greater London bounding box ─────────────────────────────
LONDON_LAT_MIN, LONDON_LAT_MAX = 51.20, 51.80
LONDON_LON_MIN, LONDON_LON_MAX = -0.60, 0.40

outside = trips[
    (trips['plat'].notna()) & (
        (trips['plat'] < LONDON_LAT_MIN) | (trips['plat'] > LONDON_LAT_MAX) |
        (trips['plon'] < LONDON_LON_MIN) | (trips['plon'] > LONDON_LON_MAX) |
        (trips['dlat'] < LONDON_LAT_MIN) | (trips['dlat'] > LONDON_LAT_MAX) |
        (trips['dlon'] < LONDON_LON_MIN) | (trips['dlon'] > LONDON_LON_MAX)
    )
]
print(f"\n[2] TRIPS WITH COORDINATES OUTSIDE GREATER LONDON BOX")
print(f"  Count: {len(outside)} ({len(outside)/len(trips)*100:.1f}% of trips)")
if len(outside) > 0:
    print("\n  Worst offenders (farthest from London):")
    outside_sorted = outside.copy()
    outside_sorted['max_lat_dev'] = (outside_sorted[['plat','dlat']].sub(51.5).abs()).max(axis=1)
    for _, r in outside_sorted.nlargest(10, 'max_lat_dev').iterrows():
        pu = (r.pickup_address or '')[:45]
        do = (r.dropoff_address or '')[:45]
        print(f"    {r.display_name:<22} £{r.trip_price_in_pound:>6.2f}  {r.distance_in_miles:>5.1f}mi")
        print(f"      FROM: {pu}")
        print(f"        TO: {do}")
        print(f"      COORDS: pickup ({r.plat:.4f},{r.plon:.4f}) dropoff ({r.dlat:.4f},{r.dlon:.4f})")

# ── 3. Distance outliers ──────────────────────────────────────────────────────
print("\n[3] DISTANCE OUTLIERS")
d_stats = trips['distance_in_miles'].describe(percentiles=[.5,.75,.9,.95,.99])
print(f"  Median:  {d_stats['50%']:.1f}mi")
print(f"  90th %:  {d_stats['90%']:.1f}mi")
print(f"  95th %:  {d_stats['95%']:.1f}mi")
print(f"  99th %:  {d_stats['99%']:.1f}mi")
print(f"  Max:     {d_stats['max']:.1f}mi")

long_trips = trips[trips['distance_in_miles'] > 30].sort_values('distance_in_miles', ascending=False)
print(f"\n  Trips over 30 miles: {len(long_trips)}")
for _, r in long_trips.head(10).iterrows():
    pu = (r.pickup_address or '')[:40]
    do = (r.dropoff_address or '')[:40]
    print(f"    {r.display_name:<22} {r.distance_in_miles:.1f}mi  £{r.trip_price_in_pound:.2f}  PZ={r.pickup_zone}  DZ={r.dropoff_zone}")
    print(f"      {pu} -> {do}")

# ── 4. Fare outliers ──────────────────────────────────────────────────────────
print("\n[4] FARE OUTLIERS")
f_stats = trips['trip_price_in_pound'].describe(percentiles=[.5,.75,.9,.95,.99])
print(f"  Median:  £{f_stats['50%']:.2f}")
print(f"  90th %:  £{f_stats['90%']:.2f}")
print(f"  95th %:  £{f_stats['95%']:.2f}")
print(f"  99th %:  £{f_stats['99%']:.2f}")
print(f"  Max:     £{f_stats['max']:.2f}")

big_fares = trips[trips['trip_price_in_pound'] > 80].sort_values('trip_price_in_pound', ascending=False)
print(f"\n  Trips over £80: {len(big_fares)}")
for _, r in big_fares.head(10).iterrows():
    pu = (r.pickup_address or '')[:40]
    do = (r.dropoff_address or '')[:40]
    print(f"    {r.display_name:<22} £{r.trip_price_in_pound:.2f}  {r.distance_in_miles:.1f}mi  PZ={r.pickup_zone}")
    print(f"      {pu} -> {do}")

# ── 5. Zero/bad coord trips ───────────────────────────────────────────────────
bad_coords = trips[trips['plat'].isna() | (trips['plat'] == 0)]
print(f"\n[5] UNPARSEABLE / ZERO COORDINATES: {len(bad_coords)} trips")
if len(bad_coords) > 0:
    sample = bad_coords[['pickup_lat_long','dropoff_latlong']].drop_duplicates().head(5)
    for _, r in sample.iterrows():
        print(f"  pickup:  [{r.pickup_lat_long}]")
        print(f"  dropoff: [{r.dropoff_latlong}]")

# ── 6. Summary: what to cut ───────────────────────────────────────────────────
print("\n[6] PROPOSED FILTER — trips to EXCLUDE from analysis:")
exclude = trips[
    (trips['plat'].notna()) & (
        (trips['plat'] < LONDON_LAT_MIN) | (trips['plat'] > LONDON_LAT_MAX) |
        (trips['plon'] < LONDON_LON_MIN) | (trips['plon'] > LONDON_LON_MAX) |
        (trips['dlat'] < LONDON_LAT_MIN) | (trips['dlat'] > LONDON_LAT_MAX) |
        (trips['dlon'] < LONDON_LON_MIN) | (trips['dlon'] > LONDON_LON_MAX) |
        (trips['distance_in_miles'] > 50)
    )
] | trips[trips['plat'].isna()]

print(f"  Total excluded: {len(exclude)} trips ({len(exclude)/len(trips)*100:.1f}%)")
print(f"  Remaining:      {len(trips)-len(exclude)} trips ({(len(trips)-len(exclude))/len(trips)*100:.1f}%)")

conn.close()
