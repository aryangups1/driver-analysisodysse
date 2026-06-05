"""
Build pickup->dropoff zone flow data for accepted vs declined trips.
Saves to parquet so the dashboard can load it fast.
"""
import psycopg2
import pandas as pd
from zones import parse_dms, assign_zone
from config import TOP_DRIVER_IDS

conn = psycopg2.connect(
    host='dev-odysse.postgres.database.azure.com', port=5432,
    dbname='unifieddwh', user='odysys', password='Odysys@2026', sslmode='require'
)
IDS = TOP_DRIVER_IDS

print("Fetching all trips (accepted + declined) with coords...")
df = pd.read_sql("""
    SELECT dim_driver_id, status,
           pickup_lat_long, dropoff_latlong,
           pickup_address,
           trip_price_in_pound, distance_in_miles,
           trips_hr, trip_booking_datetime
    FROM rep_fact_trips
    WHERE dim_driver_id = ANY(%s)
      AND pickup_lat_long IS NOT NULL AND pickup_lat_long != ''
      AND dropoff_latlong IS NOT NULL AND dropoff_latlong != ''
      AND status IN ('completed','Finished','Driver did not respond','Driver rejected')
""", conn, params=(IDS,))

print(f"  Fetched {len(df):,} rows. Parsing coordinates...")

coords_p = df["pickup_lat_long"].apply(parse_dms)
coords_d = df["dropoff_latlong"].apply(parse_dms)
df["plat"] = [c[0] for c in coords_p]
df["plon"] = [c[1] for c in coords_p]
df["dlat"] = [c[0] for c in coords_d]
df["dlon"] = [c[1] for c in coords_d]

print("  Assigning zones...")
df["pickup_zone"]  = df.apply(lambda r: assign_zone(r.plat,  r.plon), axis=1)
df["dropoff_zone"] = df.apply(lambda r: assign_zone(r.dlat, r.dlon), axis=1)

df["outcome"] = df["status"].apply(
    lambda s: "Accepted" if s in ("completed","Finished") else "Declined"
)

df = df.dropna(subset=["pickup_zone","dropoff_zone"])
df["pickup_zone"]  = df["pickup_zone"].astype(int)
df["dropoff_zone"] = df["dropoff_zone"].astype(int)
df["hour"] = pd.to_datetime(df["trip_booking_datetime"]).dt.hour

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
df[["dim_driver_id","outcome","pickup_zone","dropoff_zone",
    "trip_price_in_pound","distance_in_miles","hour","pickup_address"]].to_parquet(
    _os.path.join(_HERE, "flow_data.parquet"), index=False
)
print(f"Saved flow_data.parquet  ({len(df):,} rows)")
conn.close()
