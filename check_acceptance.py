"""
Compare accepted vs declined trips to understand what top drivers are selective about.
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

# Pull all trips across all statuses — see what they accepted vs declined
all_trips = pd.read_sql("""
    SELECT dim_driver_id, driver_full_name, status,
           pickup_lat_long, pickup_address,
           trip_price_in_pound, distance_in_miles,
           trips_hr, source,
           pickedup_trip_datetime
    FROM rep_fact_trips
    WHERE dim_driver_id = ANY(%s)
      AND pickup_lat_long IS NOT NULL AND pickup_lat_long != ''
    ORDER BY dim_driver_id, pickedup_trip_datetime
""", conn, params=(IDS,))

all_trips["display_name"] = all_trips["dim_driver_id"].map(DRIVER_NAMES)

# Parse pickup zone
coords = all_trips["pickup_lat_long"].apply(parse_dms)
all_trips["plat"] = [c[0] for c in coords]
all_trips["plon"] = [c[1] for c in coords]
all_trips["pickup_zone"] = all_trips.apply(
    lambda r: assign_zone(r.plat, r.plon), axis=1
)

# Classify statuses
ACCEPTED   = ("completed", "Finished")
DECLINED   = ("Driver did not respond", "Driver rejected", "Driver cancelled")
CANCELLED  = ("Rider cancelled", "Rider did not show")

all_trips["outcome"] = all_trips["status"].apply(
    lambda s: "Accepted" if s in ACCEPTED else
              "Declined" if s in DECLINED else
              "Rider cancelled" if s in CANCELLED else "Other"
)

print("Status breakdown across all 10 drivers:")
print(all_trips.groupby("status").size().sort_values(ascending=False).to_string())

print("\n\nAccepted vs Declined — pickup zone distribution:")
zone_outcome = all_trips[all_trips["outcome"].isin(["Accepted","Declined"])].groupby(
    ["outcome","pickup_zone"]
).size().unstack(fill_value=0)
zone_pct = (zone_outcome.div(zone_outcome.sum(axis=1), axis=0) * 100).round(1)
print(zone_pct.to_string())

print("\n\nAccepted vs Declined — hour of day distribution:")
hour_outcome = all_trips[all_trips["outcome"].isin(["Accepted","Declined"])].groupby(
    ["outcome","trips_hr"]
).size().unstack(fill_value=0)
hour_pct = (hour_outcome.div(hour_outcome.sum(axis=1), axis=0) * 100).round(1)
print(hour_pct.to_string())

print("\n\nAccepted vs Declined — trip distance (accepted only has distance):")
accepted = all_trips[all_trips["outcome"]=="Accepted"]["distance_in_miles"]
print(f"  Accepted trips: mean {accepted.mean():.1f}mi  median {accepted.median():.1f}mi")
print(f"  Short (<3mi):  {(accepted < 3).mean()*100:.1f}% of accepted trips")
print(f"  Long (>10mi): {(accepted > 10).mean()*100:.1f}% of accepted trips")

print("\n\nPer driver — acceptance rate and what zone they decline most from:")
for did in IDS:
    d = all_trips[all_trips["dim_driver_id"]==did]
    name = DRIVER_NAMES.get(did, str(did))
    total = len(d[d["outcome"].isin(["Accepted","Declined"])])
    accepted_n = len(d[d["outcome"]=="Accepted"])
    accept_rate = accepted_n / total * 100 if total > 0 else 0

    # Most declined zone
    declined = d[d["outcome"]=="Declined"]
    if len(declined) > 0:
        top_declined_zone = declined["pickup_zone"].value_counts().idxmax()
        top_declined_zone_pct = declined["pickup_zone"].value_counts(normalize=True).max() * 100
    else:
        top_declined_zone, top_declined_zone_pct = "N/A", 0

    # Most accepted zone
    accepted_d = d[d["outcome"]=="Accepted"]
    if len(accepted_d) > 0:
        top_accepted_zone = accepted_d["pickup_zone"].value_counts().idxmax()
    else:
        top_accepted_zone = "N/A"

    print(f"  {name:<28}  accept={accept_rate:.0f}%  most declined from Z{top_declined_zone} ({top_declined_zone_pct:.0f}%)  most accepted from Z{top_accepted_zone}")

print("\n\nSource breakdown — do they decline differently by platform?")
src_outcome = all_trips[all_trips["outcome"].isin(["Accepted","Declined"])].groupby(
    ["source","outcome"]
).size().unstack(fill_value=0)
src_outcome["accept_rate"] = (src_outcome.get("Accepted",0) /
                               src_outcome.sum(axis=1) * 100).round(1)
print(src_outcome.to_string())

conn.close()
