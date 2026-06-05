import psycopg2
from zones import parse_dms, assign_zone

conn = psycopg2.connect(
    host='dev-odysse.postgres.database.azure.com',
    port=5432, dbname='unifieddwh',
    user='odysys', password='Odysys@2026', sslmode='require'
)
cur = conn.cursor()
cur.execute("""
    SELECT pickup_lat_long, dropoff_latlong, pickup_address, dropoff_address
    FROM rep_fact_trips
    WHERE dim_driver_id IN (81,128,123) AND status='completed'
      AND pickup_lat_long IS NOT NULL LIMIT 12
""")
rows = cur.fetchall()
cur.close()
conn.close()

ok, failed = 0, 0
for pickup_ll, dropoff_ll, pickup_addr, dropoff_addr in rows:
    plat, plon = parse_dms(pickup_ll)
    dlat, dlon = parse_dms(dropoff_ll)
    pzone = assign_zone(plat, plon)
    dzone = assign_zone(dlat, dlon)
    addr = (pickup_addr or '')[:35].encode('ascii', errors='replace').decode()
    fmt = pickup_ll.encode('ascii', errors='replace').decode() if pickup_ll else ''
    result = f"Z{pzone}->Z{dzone}  {addr}"
    print(result)
    if pzone and dzone:
        ok += 1
    else:
        failed += 1
        print(f"  FAILED to parse: [{fmt}]")

print(f"\nOK: {ok}  Failed: {failed}")
